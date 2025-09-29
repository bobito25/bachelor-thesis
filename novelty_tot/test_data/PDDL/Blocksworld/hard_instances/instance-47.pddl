

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b a)
(on c d)
(on d e)
(on e b)
(ontable f)
(clear c)
(clear f)
)
(:goal
(and
(on a b)
(on b d)
(on d e))
)
)


