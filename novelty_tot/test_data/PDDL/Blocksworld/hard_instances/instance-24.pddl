

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(ontable b)
(on c e)
(on d c)
(on e f)
(on f a)
(clear d)
)
(:goal
(and
(on a d)
(on b c)
(on e b))
)
)


