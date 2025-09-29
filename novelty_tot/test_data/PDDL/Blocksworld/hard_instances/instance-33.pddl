

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(on b a)
(on c d)
(ontable d)
(on e f)
(on f c)
(clear b)
)
(:goal
(and
(on c d)
(on d e)
(on e b))
)
)


